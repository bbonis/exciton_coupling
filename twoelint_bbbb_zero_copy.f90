! file: twoelint_bbbb_zero_copy.f90
!
! Fortran kernel for calc_BBBB (zero-copy).
! Matches corrected Python logic:
!   32: ACA_B_right * CS_EA_A * AC_IP_B_left
!   24: ACA_B_left  * CA_EA_B_right * SC_IP_A
!   34: AC_IP_B_left * CS_EA_A * CA_EA_B_right * SC_IP_A
module twoelint_bbbb_mod
  use twoelint_core_mod, only: gen_perms, u16_from_i16, get2
  implicit none
  private
  public :: bbbb_accum_all

contains

  subroutine bbbb_accum_all(pos1, pos2, pos3, pos4, a, b, c, d, v, threshold, &
                            aca_b_right, aca_b_right_ncol, &
                            aca_b_left,  aca_b_left_ncol, &
                            cs_ea_a,     cs_ea_a_ncol, &
                            ac_ip_b_left, ac_ip_b_left_ncol, &
                            ca_ea_b_right, ca_ea_b_right_ncol, &
                            sc_ip_a, sc_ip_a_ncol, &
                            out32c, out32x, out24c, out24x, out34c, out34x)
    implicit none
    integer*8, intent(in) :: pos1(:), pos2(:), pos3(:), pos4(:)
    integer*2, intent(in) :: a(:), b(:), c(:), d(:)
    double precision, intent(in) :: v(:)
    integer, intent(in) :: threshold

    double precision, intent(in) :: aca_b_right(:), aca_b_left(:), cs_ea_a(:)
    double precision, intent(in) :: ac_ip_b_left(:), ca_ea_b_right(:), sc_ip_a(:)
    integer, intent(in) :: aca_b_right_ncol, aca_b_left_ncol, cs_ea_a_ncol
    integer, intent(in) :: ac_ip_b_left_ncol, ca_ea_b_right_ncol, sc_ip_a_ncol

    double precision, intent(out) :: out32c, out32x, out24c, out24x, out34c, out34x

    out32c = 0.0d0; out32x = 0.0d0
    out24c = 0.0d0; out24x = 0.0d0
    out34c = 0.0d0; out34x = 0.0d0

    call accum_pos_block(pos1, 1, a, b, c, d, v, threshold, &
                         aca_b_right, aca_b_right_ncol, aca_b_left, aca_b_left_ncol, &
                         cs_ea_a, cs_ea_a_ncol, ac_ip_b_left, ac_ip_b_left_ncol, &
                         ca_ea_b_right, ca_ea_b_right_ncol, sc_ip_a, sc_ip_a_ncol, &
                         out32c, out32x, out24c, out24x, out34c, out34x)

    call accum_pos_block(pos2, 2, a, b, c, d, v, threshold, &
                         aca_b_right, aca_b_right_ncol, aca_b_left, aca_b_left_ncol, &
                         cs_ea_a, cs_ea_a_ncol, ac_ip_b_left, ac_ip_b_left_ncol, &
                         ca_ea_b_right, ca_ea_b_right_ncol, sc_ip_a, sc_ip_a_ncol, &
                         out32c, out32x, out24c, out24x, out34c, out34x)

    call accum_pos_block(pos3, 3, a, b, c, d, v, threshold, &
                         aca_b_right, aca_b_right_ncol, aca_b_left, aca_b_left_ncol, &
                         cs_ea_a, cs_ea_a_ncol, ac_ip_b_left, ac_ip_b_left_ncol, &
                         ca_ea_b_right, ca_ea_b_right_ncol, sc_ip_a, sc_ip_a_ncol, &
                         out32c, out32x, out24c, out24x, out34c, out34x)

    call accum_pos_block(pos4, 4, a, b, c, d, v, threshold, &
                         aca_b_right, aca_b_right_ncol, aca_b_left, aca_b_left_ncol, &
                         cs_ea_a, cs_ea_a_ncol, ac_ip_b_left, ac_ip_b_left_ncol, &
                         ca_ea_b_right, ca_ea_b_right_ncol, sc_ip_a, sc_ip_a_ncol, &
                         out32c, out32x, out24c, out24x, out34c, out34x)

  end subroutine bbbb_accum_all


  subroutine accum_pos_block(pos, ucount, a, b, c, d, v, threshold, &
                             aca_b_right, aca_b_right_ncol, aca_b_left, aca_b_left_ncol, &
                             cs_ea_a, cs_ea_a_ncol, ac_ip_b_left, ac_ip_b_left_ncol, &
                             ca_ea_b_right, ca_ea_b_right_ncol, sc_ip_a, sc_ip_a_ncol, &
                             out32c, out32x, out24c, out24x, out34c, out34x)
    implicit none
    integer*8, intent(in) :: pos(:)
    integer, intent(in) :: ucount, threshold
    integer*2, intent(in) :: a(:), b(:), c(:), d(:)
    double precision, intent(in) :: v(:)

    double precision, intent(in) :: aca_b_right(:), aca_b_left(:), cs_ea_a(:)
    double precision, intent(in) :: ac_ip_b_left(:), ca_ea_b_right(:), sc_ip_a(:)
    integer, intent(in) :: aca_b_right_ncol, aca_b_left_ncol, cs_ea_a_ncol
    integer, intent(in) :: ac_ip_b_left_ncol, ca_ea_b_right_ncol, sc_ip_a_ncol

    double precision, intent(inout) :: out32c, out32x, out24c, out24x, out34c, out34x

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

      i = u16_from_i16(a(p1)) - 1 - threshold
      j = u16_from_i16(b(p1)) - 1 - threshold
      k = u16_from_i16(c(p1)) - 1 - threshold
      l = u16_from_i16(d(p1)) - 1 - threshold

      value = v(p1)

      call gen_perms(ucount, i, j, k, l, nperm, ii, jj, kk, ll)

      do q = 1, nperm
        out32c = out32c + Cpref * get2(aca_b_right, aca_b_right_ncol, ii(q), jj(q)) * &
                 get2(cs_ea_a, cs_ea_a_ncol, 0, kk(q)) * &
                 get2(ac_ip_b_left, ac_ip_b_left_ncol, ll(q), 0) * value

        out32x = out32x + Xpref * get2(aca_b_right, aca_b_right_ncol, ii(q), kk(q)) * &
                 get2(cs_ea_a, cs_ea_a_ncol, 0, ll(q)) * &
                 get2(ac_ip_b_left, ac_ip_b_left_ncol, jj(q), 0) * value

        out24c = out24c + Cpref * get2(aca_b_left, aca_b_left_ncol, ii(q), jj(q)) * &
                 get2(ca_ea_b_right, ca_ea_b_right_ncol, 0, kk(q)) * &
                 get2(sc_ip_a, sc_ip_a_ncol, ll(q), 0) * value

        out24x = out24x + Xpref * get2(aca_b_left, aca_b_left_ncol, ii(q), kk(q)) * &
                 get2(ca_ea_b_right, ca_ea_b_right_ncol, 0, ll(q)) * &
                 get2(sc_ip_a, sc_ip_a_ncol, jj(q), 0) * value

        out34c = out34c + Cpref * get2(ac_ip_b_left, ac_ip_b_left_ncol, ii(q), 0) * &
                 get2(cs_ea_a, cs_ea_a_ncol, 0, jj(q)) * &
                 get2(ca_ea_b_right, ca_ea_b_right_ncol, 0, kk(q)) * &
                 get2(sc_ip_a, sc_ip_a_ncol, ll(q), 0) * value

        out34x = out34x + Xpref * get2(ac_ip_b_left, ac_ip_b_left_ncol, ii(q), 0) * &
                 get2(cs_ea_a, cs_ea_a_ncol, 0, kk(q)) * &
                 get2(ca_ea_b_right, ca_ea_b_right_ncol, 0, ll(q)) * &
                 get2(sc_ip_a, sc_ip_a_ncol, jj(q), 0) * value
      end do
    end do

  end subroutine accum_pos_block

end module twoelint_bbbb_mod
