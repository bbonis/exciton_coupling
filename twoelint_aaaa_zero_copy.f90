! file: twoelint_aaaa_zero_copy.f90
!
! Fortran kernel for calc_AAAA (zero-copy).
! Matches corrected Python logic:
!   41: ACA_A_right * CS_EA_B * AC_IP_A_left
!   13: ACA_A_left  * CA_EA_A_right * SC_IP_B
!   43: AC_IP_A_left * CS_EA_B * CA_EA_A_right * SC_IP_B
module twoelint_aaaa_mod
  use twoelint_core_mod, only: gen_perms, u16_from_i16, get2
  implicit none
  private
  public :: aaaa_accum_all

contains

  subroutine aaaa_accum_all(pos1, pos2, pos3, pos4, a, b, c, d, v, threshold, &
                            aca_a_right, aca_a_right_ncol, &
                            aca_a_left,  aca_a_left_ncol, &
                            cs_ea_b,     cs_ea_b_ncol, &
                            ac_ip_a_left, ac_ip_a_left_ncol, &
                            ca_ea_a_right, ca_ea_a_right_ncol, &
                            sc_ip_b, sc_ip_b_ncol, &
                            out41c, out41x, out13c, out13x, out43c, out43x)
    implicit none
    integer*8, intent(in) :: pos1(:), pos2(:), pos3(:), pos4(:)
    integer*2, intent(in) :: a(:), b(:), c(:), d(:)
    double precision, intent(in) :: v(:)
    integer, intent(in) :: threshold

    double precision, intent(in) :: aca_a_right(:), aca_a_left(:), cs_ea_b(:)
    double precision, intent(in) :: ac_ip_a_left(:), ca_ea_a_right(:), sc_ip_b(:)
    integer, intent(in) :: aca_a_right_ncol, aca_a_left_ncol, cs_ea_b_ncol
    integer, intent(in) :: ac_ip_a_left_ncol, ca_ea_a_right_ncol, sc_ip_b_ncol

    double precision, intent(out) :: out41c, out41x, out13c, out13x, out43c, out43x

    out41c = 0.0d0; out41x = 0.0d0
    out13c = 0.0d0; out13x = 0.0d0
    out43c = 0.0d0; out43x = 0.0d0

    call accum_pos_block(pos1, 1, a, b, c, d, v, threshold, &
                         aca_a_right, aca_a_right_ncol, aca_a_left, aca_a_left_ncol, &
                         cs_ea_b, cs_ea_b_ncol, ac_ip_a_left, ac_ip_a_left_ncol, &
                         ca_ea_a_right, ca_ea_a_right_ncol, sc_ip_b, sc_ip_b_ncol, &
                         out41c, out41x, out13c, out13x, out43c, out43x)

    call accum_pos_block(pos2, 2, a, b, c, d, v, threshold, &
                         aca_a_right, aca_a_right_ncol, aca_a_left, aca_a_left_ncol, &
                         cs_ea_b, cs_ea_b_ncol, ac_ip_a_left, ac_ip_a_left_ncol, &
                         ca_ea_a_right, ca_ea_a_right_ncol, sc_ip_b, sc_ip_b_ncol, &
                         out41c, out41x, out13c, out13x, out43c, out43x)

    call accum_pos_block(pos3, 3, a, b, c, d, v, threshold, &
                         aca_a_right, aca_a_right_ncol, aca_a_left, aca_a_left_ncol, &
                         cs_ea_b, cs_ea_b_ncol, ac_ip_a_left, ac_ip_a_left_ncol, &
                         ca_ea_a_right, ca_ea_a_right_ncol, sc_ip_b, sc_ip_b_ncol, &
                         out41c, out41x, out13c, out13x, out43c, out43x)

    call accum_pos_block(pos4, 4, a, b, c, d, v, threshold, &
                         aca_a_right, aca_a_right_ncol, aca_a_left, aca_a_left_ncol, &
                         cs_ea_b, cs_ea_b_ncol, ac_ip_a_left, ac_ip_a_left_ncol, &
                         ca_ea_a_right, ca_ea_a_right_ncol, sc_ip_b, sc_ip_b_ncol, &
                         out41c, out41x, out13c, out13x, out43c, out43x)

  end subroutine aaaa_accum_all


  subroutine accum_pos_block(pos, ucount, a, b, c, d, v, threshold, &
                             aca_a_right, aca_a_right_ncol, aca_a_left, aca_a_left_ncol, &
                             cs_ea_b, cs_ea_b_ncol, ac_ip_a_left, ac_ip_a_left_ncol, &
                             ca_ea_a_right, ca_ea_a_right_ncol, sc_ip_b, sc_ip_b_ncol, &
                             out41c, out41x, out13c, out13x, out43c, out43x)
    implicit none
    integer*8, intent(in) :: pos(:)
    integer, intent(in) :: ucount
    integer*2, intent(in) :: a(:), b(:), c(:), d(:)
    double precision, intent(in) :: v(:)
    integer, intent(in) :: threshold

    double precision, intent(in) :: aca_a_right(:), aca_a_left(:), cs_ea_b(:)
    double precision, intent(in) :: ac_ip_a_left(:), ca_ea_a_right(:), sc_ip_b(:)
    integer, intent(in) :: aca_a_right_ncol, aca_a_left_ncol, cs_ea_b_ncol
    integer, intent(in) :: ac_ip_a_left_ncol, ca_ea_a_right_ncol, sc_ip_b_ncol

    double precision, intent(inout) :: out41c, out41x, out13c, out13x, out43c, out43x

    integer*8 :: t, p1
    integer :: i, j, k, l
    integer :: ii(8), jj(8), kk(8), ll(8)
    integer :: nperm, q
    double precision :: value
    double precision, parameter :: Cpref = 4.0d0
    double precision, parameter :: Xpref = -2.0d0

    if (size(pos) == 0) return

    do t = 1_8, int(size(pos), 8)
      p1 = pos(t) + 1_8

      i = u16_from_i16(a(p1)) - 1
      j = u16_from_i16(b(p1)) - 1
      k = u16_from_i16(c(p1)) - 1
      l = u16_from_i16(d(p1)) - 1



      if (i >= threshold .or. j >= threshold .or. k >= threshold .or. l >= threshold) then
        i = i - threshold
        j = j - threshold
        k = k - threshold
        l = l - threshold
      end if
      value = v(p1)

      call gen_perms(ucount, i, j, k, l, nperm, ii, jj, kk, ll)

      do q = 1, nperm
        out41c = out41c + Cpref * get2(aca_a_right, aca_a_right_ncol, ii(q), jj(q)) * &
                 get2(cs_ea_b, cs_ea_b_ncol, 0, kk(q)) * &
                 get2(ac_ip_a_left, ac_ip_a_left_ncol, ll(q), 0) * value

        out41x = out41x + Xpref * get2(aca_a_right, aca_a_right_ncol, ii(q), kk(q)) * &
                 get2(cs_ea_b, cs_ea_b_ncol, 0, ll(q)) * &
                 get2(ac_ip_a_left, ac_ip_a_left_ncol, jj(q), 0) * value

        out13c = out13c + Cpref * get2(aca_a_left, aca_a_left_ncol, ii(q), jj(q)) * &
                 get2(ca_ea_a_right, ca_ea_a_right_ncol, 0, kk(q)) * &
                 get2(sc_ip_b, sc_ip_b_ncol, ll(q), 0) * value

        out13x = out13x + Xpref * get2(aca_a_left, aca_a_left_ncol, ii(q), kk(q)) * &
                 get2(ca_ea_a_right, ca_ea_a_right_ncol, 0, ll(q)) * &
                 get2(sc_ip_b, sc_ip_b_ncol, jj(q), 0) * value

        out43c = out43c + Cpref * get2(ac_ip_a_left, ac_ip_a_left_ncol, ii(q), 0) * &
                 get2(cs_ea_b, cs_ea_b_ncol, 0, jj(q)) * &
                 get2(ca_ea_a_right, ca_ea_a_right_ncol, 0, kk(q)) * &
                 get2(sc_ip_b, sc_ip_b_ncol, ll(q), 0) * value

        out43x = out43x + Xpref * get2(ac_ip_a_left, ac_ip_a_left_ncol, ii(q), 0) * &
                 get2(cs_ea_b, cs_ea_b_ncol, 0, kk(q)) * &
                 get2(ca_ea_a_right, ca_ea_a_right_ncol, 0, ll(q)) * &
                 get2(sc_ip_b, sc_ip_b_ncol, jj(q), 0) * value
      end do
    end do

  end subroutine accum_pos_block

end module twoelint_aaaa_mod
