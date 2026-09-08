! file: twoelint_baaa_zero_copy.f90
!
! Zero-copy Fortran kernel for BAAA contributions.
! - Accepts pos arrays (0-based) into shared backing arrays a/b/c/d/v.
! - Reads uint16 indices via int16 bit-view + repair (x<0 => x+65536).
! - Replicates Python calc_BAAA + calc_term_BAAA_{2,3,4} logic exactly.
!
module twoelint_baaa_mod
  use twoelint_core_mod, only: u16_from_i16

  implicit none
  private
  public :: baaa_accum_all

contains

  subroutine baaa_accum_all( &
      pos1, pos2, pos3, pos4, &
      a, b, c, d, v, nbas_a, &
      sc_b_ab, sc_b_ab_ncol, &
      cs_b_ba, cs_b_ba_ncol, &
      sc_a_ba, sc_a_ba_ncol, &
      cs_a_ab, cs_a_ab_ncol, &
      sc_b_ab_pa, sc_b_ab_pa_ncol, &
      sc_a_ba_pa, sc_a_ba_pa_ncol, &
      cs_a_ab_iq, cs_a_ab_iq_ncol, &
      cs_ea_a, cs_ea_a_ncol, &
      sc_ip_a, sc_ip_a_ncol, &
      sc_ip_b_p, sc_ip_b_p_ncol, &
      cs_ea_b_q, cs_ea_b_q_ncol, &
      scs_b_aa_pa, scs_b_aa_pa_ncol, &
      ac_ip_b_left, ac_ip_b_left_ncol, &
      ca_ea_b_left, ca_ea_b_left_ncol, &
      ac_ip_b_right, ac_ip_b_right_ncol, &
      ca_ea_b_right, ca_ea_b_right_ncol, &
      sc_ip_b, sc_ip_b_ncol, &
      cs_ea_b, cs_ea_b_ncol, &
      aca_a_right, aca_a_right_ncol, &
      aca_a_left, aca_a_left_ncol, &
      ca_ea_a_right, ca_ea_a_right_ncol, &
      ac_ip_a_right, ac_ip_a_right_ncol, &
      ca_ea_a_left, ca_ea_a_left_ncol, &
      ac_ip_a_left, ac_ip_a_left_ncol, &
      out31c, out31x, out41c, out41x, out13c, out13x, out14c, out14x, &
      out21c, out21x, out12c, out12x, out43c_csea, out43x_csea, out43c_scip, out43x_scip, &
      out42c, out42x, out41c_2s, out41x_2s, out23c_2s_ip, out23x_2s_ip, out13c_2s_ip, out13x_2s_ip, &
      out21c_1s_occ_arb, out21x_1s_occ_arb, &
      out31c_2s_ea_ip_arb, out31x_2s_ea_ip_arb, &
      out31c_2s_occ_ip_arb, out31x_2s_occ_ip_arb, &
      out14c_2s_ip_ea_arb, out14x_2s_ip_ea_arb, &
      out21c_3s_a_occ_b_virt_b_occ_arb, out21x_3s_a_occ_b_virt_b_occ_arb, &
      out14c_2s_occ_arb_ea_arb, out14x_2s_occ_arb_ea_arb, &
      out21c_3s_b_virt_b_occ_arb_a_virt_arb, out21x_3s_b_virt_b_occ_arb_a_virt_arb, &
      out14c_2s_virt_ea_arb, out14x_2s_virt_ea_arb, &
      err_flag)

    implicit none

    integer*8, intent(in) :: pos1(:), pos2(:), pos3(:), pos4(:)
    integer*2, intent(in) :: a(*), b(*), c(*), d(*)
    double precision, intent(in) :: v(*)
    integer, intent(in) :: nbas_a

    double precision, intent(in) :: sc_b_ab(*), cs_b_ba(*), sc_a_ba(*), cs_a_ab(*)
    double precision, intent(in) :: sc_b_ab_pa(*), sc_a_ba_pa(*), cs_a_ab_iq(*)
    double precision, intent(in) :: cs_ea_a(*), sc_ip_a(*), sc_ip_b_p(*), cs_ea_b_q(*), scs_b_aa_pa(*)
    double precision, intent(in) :: ac_ip_b_left(*), ca_ea_b_left(*), ac_ip_b_right(*), ca_ea_b_right(*)
    double precision, intent(in) :: sc_ip_b(*), cs_ea_b(*)
    double precision, intent(in) :: aca_a_right(*), aca_a_left(*)
    double precision, intent(in) :: ca_ea_a_right(*), ac_ip_a_right(*), ca_ea_a_left(*), ac_ip_a_left(*)

    integer, intent(in) :: sc_b_ab_ncol, cs_b_ba_ncol, sc_a_ba_ncol, cs_a_ab_ncol
    integer, intent(in) :: sc_b_ab_pa_ncol, sc_a_ba_pa_ncol, cs_a_ab_iq_ncol
    integer, intent(in) :: cs_ea_a_ncol, sc_ip_a_ncol, sc_ip_b_p_ncol, cs_ea_b_q_ncol, scs_b_aa_pa_ncol
    integer, intent(in) :: ac_ip_b_left_ncol, ca_ea_b_left_ncol, ac_ip_b_right_ncol, ca_ea_b_right_ncol
    integer, intent(in) :: sc_ip_b_ncol, cs_ea_b_ncol
    integer, intent(in) :: aca_a_right_ncol, aca_a_left_ncol
    integer, intent(in) :: ca_ea_a_right_ncol, ac_ip_a_right_ncol, ca_ea_a_left_ncol, ac_ip_a_left_ncol

    double precision, intent(out) :: out31c, out31x, out41c, out41x, out13c, out13x, out14c, out14x
    double precision, intent(out) :: out21c, out21x, out12c, out12x, out43c_csea, out43x_csea, out43c_scip, out43x_scip
    double precision, intent(out) :: out42c, out42x, out41c_2s, out41x_2s, out23c_2s_ip, out23x_2s_ip, out13c_2s_ip, out13x_2s_ip
    double precision, intent(out) :: out21c_1s_occ_arb, out21x_1s_occ_arb
    double precision, intent(out) :: out31c_2s_ea_ip_arb, out31x_2s_ea_ip_arb
    double precision, intent(out) :: out31c_2s_occ_ip_arb, out31x_2s_occ_ip_arb
    double precision, intent(out) :: out14c_2s_ip_ea_arb, out14x_2s_ip_ea_arb
    double precision, intent(out) :: out21c_3s_a_occ_b_virt_b_occ_arb, out21x_3s_a_occ_b_virt_b_occ_arb
    double precision, intent(out) :: out14c_2s_occ_arb_ea_arb, out14x_2s_occ_arb_ea_arb
    double precision, intent(out) :: out21c_3s_b_virt_b_occ_arb_a_virt_arb, out21x_3s_b_virt_b_occ_arb_a_virt_arb
    double precision, intent(out) :: out14c_2s_virt_ea_arb, out14x_2s_virt_ea_arb
    integer, intent(out) :: err_flag

    double precision, parameter :: Cpref = 4.0d0
    double precision, parameter :: Xpref = -2.0d0

    integer*8 :: t, p1
    integer :: i, j, k, l
    integer :: ai(3), bi
    integer :: ok
    double precision :: val
    logical :: is_type_1

    out31c = 0.0d0; out31x = 0.0d0
    out41c = 0.0d0; out41x = 0.0d0
    out13c = 0.0d0; out13x = 0.0d0
    out14c = 0.0d0; out14x = 0.0d0
    out21c = 0.0d0; out21x = 0.0d0
    out12c = 0.0d0; out12x = 0.0d0
    out43c_csea = 0.0d0; out43x_csea = 0.0d0
    out43c_scip = 0.0d0; out43x_scip = 0.0d0
    out42c = 0.0d0; out42x = 0.0d0
    out41c_2s = 0.0d0; out41x_2s = 0.0d0
    out23c_2s_ip = 0.0d0; out23x_2s_ip = 0.0d0
    out13c_2s_ip = 0.0d0; out13x_2s_ip = 0.0d0
    out21c_1s_occ_arb = 0.0d0; out21x_1s_occ_arb = 0.0d0
    out31c_2s_ea_ip_arb = 0.0d0; out31x_2s_ea_ip_arb = 0.0d0
    out31c_2s_occ_ip_arb = 0.0d0; out31x_2s_occ_ip_arb = 0.0d0
    out14c_2s_ip_ea_arb = 0.0d0; out14x_2s_ip_ea_arb = 0.0d0
    out21c_3s_a_occ_b_virt_b_occ_arb = 0.0d0; out21x_3s_a_occ_b_virt_b_occ_arb = 0.0d0
    out14c_2s_occ_arb_ea_arb = 0.0d0; out14x_2s_occ_arb_ea_arb = 0.0d0
    out21c_3s_b_virt_b_occ_arb_a_virt_arb = 0.0d0; out21x_3s_b_virt_b_occ_arb_a_virt_arb = 0.0d0
    out14c_2s_virt_ea_arb = 0.0d0; out14x_2s_virt_ea_arb = 0.0d0

    err_flag = 0

    ! Process in the same effective order as Python dict insertion (1,2,3,4):
    ! - BAAA only uses 2/3/4, but we accept pos1 for completeness and treat it as the "else" path.
    call loop_u2(pos1)
    call loop_u2(pos2)
    call loop_u3(pos3)
    call loop_u4(pos4)

  contains

    double precision function get2(A, ii, jj, ncol)
      double precision, intent(in) :: A(*)
      integer, intent(in) :: ii, jj, ncol
      get2 = A(ii*ncol + jj + 1)
    end function get2

    ! Contractions over the arbitrary/root index used by the post-original
    ! BAAA terms.  The Python arrays have a singleton root dimension in the
    ! present workflow; keeping the loop here also handles a larger compatible
    ! root dimension without changing the zero-copy interface.
    double precision function dot_ipr_eabq(a_row, a_col)
      integer, intent(in) :: a_row, a_col
      integer :: q
      dot_ipr_eabq = 0.0d0
      do q = 0, ac_ip_a_right_ncol - 1
        dot_ipr_eabq = dot_ipr_eabq + get2(ac_ip_a_right, a_row, q, ac_ip_a_right_ncol) &
             & * get2(cs_ea_b_q, q, a_col, cs_ea_b_q_ncol)
      end do
    end function dot_ipr_eabq

    double precision function dot_ipbp_eaa(a_row, b_col)
      integer, intent(in) :: a_row, b_col
      integer :: q
      dot_ipbp_eaa = 0.0d0
      do q = 0, sc_ip_b_p_ncol - 1
        dot_ipbp_eaa = dot_ipbp_eaa + get2(sc_ip_b_p, a_row, q, sc_ip_b_p_ncol) &
             & * get2(cs_ea_a, q, b_col, cs_ea_a_ncol)
      end do
    end function dot_ipbp_eaa

    double precision function dot_eala_ipbp(a_col, a_row)
      integer, intent(in) :: a_col, a_row
      integer :: q
      dot_eala_ipbp = 0.0d0
      do q = 0, sc_ip_b_p_ncol - 1
        dot_eala_ipbp = dot_eala_ipbp + get2(ca_ea_a_left, q, a_col, ca_ea_a_left_ncol) &
             & * get2(sc_ip_b_p, a_row, q, sc_ip_b_p_ncol)
      end do
    end function dot_eala_ipbp

    double precision function dot_eabq_ipa(a_col, b_row)
      integer, intent(in) :: a_col, b_row
      integer :: q
      dot_eabq_ipa = 0.0d0
      do q = 0, sc_ip_a_ncol - 1
        dot_eabq_ipa = dot_eabq_ipa + get2(cs_ea_b_q, q, a_col, cs_ea_b_q_ncol) &
             & * get2(sc_ip_a, b_row, q, sc_ip_a_ncol)
      end do
    end function dot_eabq_ipa

    subroutine split_ab(i1, j1, k1, l1, a_out, b_out, ok_out)
      integer, intent(in) :: i1, j1, k1, l1
      integer, intent(out) :: a_out(3), b_out
      integer, intent(out) :: ok_out
      integer :: cntA
      integer :: x

      cntA = 0
      b_out = -1

      x = i1
      if (x <= nbas_a) then
        cntA = cntA + 1
        a_out(cntA) = x - 1
      else
        b_out = x - 1 - nbas_a
      end if

      x = j1
      if (x <= nbas_a) then
        cntA = cntA + 1
        a_out(cntA) = x - 1
      else
        b_out = x - 1 - nbas_a
      end if

      x = k1
      if (x <= nbas_a) then
        cntA = cntA + 1
        a_out(cntA) = x - 1
      else
        b_out = x - 1 - nbas_a
      end if

      x = l1
      if (x <= nbas_a) then
        cntA = cntA + 1
        a_out(cntA) = x - 1
      else
        b_out = x - 1 - nbas_a
      end if

      if (cntA == 3 .and. b_out >= 0) then
        ok_out = 1
      else
        ok_out = 0
      end if
    end subroutine split_ab

    subroutine loop_u2(pos)
      integer*8, intent(in) :: pos(:)
      do t = 1, size(pos)
        p1 = pos(t) + 1_8
        i = u16_from_i16(a(p1))
        j = u16_from_i16(b(p1))
        k = u16_from_i16(c(p1))
        l = u16_from_i16(d(p1))
        val = v(p1)

        call split_ab(i, j, k, l, ai, bi, ok)
        if (ok == 1) then
          call accum_u2(ai, bi, val)
        end if
      end do
    end subroutine loop_u2

    subroutine loop_u3(pos)
      integer*8, intent(in) :: pos(:)
      integer :: k0, l0
      do t = 1, size(pos)
        p1 = pos(t) + 1_8
        i = u16_from_i16(a(p1))
        j = u16_from_i16(b(p1))
        k = u16_from_i16(c(p1))
        l = u16_from_i16(d(p1))
        val = v(p1)

        k0 = k
        l0 = l
        is_type_1 = (k0 /= l0)

        if (j == k) then
          k = l
          l = k0
        end if

        call split_ab(i, j, k, l, ai, bi, ok)
        if (ok == 1) then
          call accum_u3(ai, bi, val, is_type_1)
        end if
      end do
    end subroutine loop_u3

    subroutine loop_u4(pos)
      integer*8, intent(in) :: pos(:)
      logical :: swap, bad
      integer :: k0
      do t = 1, size(pos)
        p1 = pos(t) + 1_8
        i = u16_from_i16(a(p1))
        j = u16_from_i16(b(p1))
        k = u16_from_i16(c(p1))
        l = u16_from_i16(d(p1))
        val = v(p1)

        swap = (j < k) .and. (j <= l)
        bad  = (j >= k) .and. (k < l)
        if (bad) then
          err_flag = 1
          return
        end if
        if (swap) then
          k0 = k
          k = l
          l = k0
        end if

        call split_ab(i, j, k, l, ai, bi, ok)
        if (ok == 1) then
          call accum_u4(ai, bi, val)
        end if
      end do
    end subroutine loop_u4

    subroutine accum_u2(idxA, idxB, value)
      integer, intent(in) :: idxA(3), idxB
      double precision, intent(in) :: value
      integer :: a0, a1, a2
      double precision :: term_31, term_41, term_13, term_14, term_43_ea, term_43_ip

      a0 = idxA(1); a1 = idxA(2); a2 = idxA(3)

      term_31    = value * get2(ac_ip_b_left,  idxB, 0, ac_ip_b_left_ncol)
      term_41    = value * get2(ca_ea_b_left,  0, idxB, ca_ea_b_left_ncol)
      term_13    = value * get2(ac_ip_b_right, idxB, 0, ac_ip_b_right_ncol)
      term_14    = value * get2(ca_ea_b_right, 0, idxB, ca_ea_b_right_ncol)
      term_43_ea = value * get2(ca_ea_b_left,  0, idxB, ca_ea_b_left_ncol)
      term_43_ip = value * get2(ac_ip_b_right, idxB, 0, ac_ip_b_right_ncol)

      out12x = out12x + Xpref * get2(aca_a_left, a0, a1, aca_a_left_ncol) * get2(sc_b_ab, a2, idxB, sc_b_ab_ncol) * value
      out12c = out12c + Cpref * get2(aca_a_left, a0, a1, aca_a_left_ncol) * get2(sc_b_ab, a2, idxB, sc_b_ab_ncol) * value

      out21x = out21x + Xpref * get2(aca_a_right, a0, a1, aca_a_right_ncol) * get2(cs_b_ba, idxB, a2, cs_b_ba_ncol) * value
      out21c = out21c + Cpref * get2(aca_a_right, a0, a1, aca_a_right_ncol) * get2(cs_b_ba, idxB, a2, cs_b_ba_ncol) * value

      out42x = out42x + Xpref * get2(sc_b_ab, a0, idxB, sc_b_ab_ncol) * get2(cs_ea_b, 0, a1, cs_ea_b_ncol) &
           & * get2(ac_ip_a_left, a2, 0, ac_ip_a_left_ncol) * value
      out42c = out42c + Cpref * get2(sc_b_ab, a0, idxB, sc_b_ab_ncol) * get2(cs_ea_b, 0, a1, cs_ea_b_ncol) &
           & * get2(ac_ip_a_left, a2, 0, ac_ip_a_left_ncol) * value

      out41x = out41x + Xpref * get2(aca_a_right, a0, a1, aca_a_right_ncol) * get2(ac_ip_a_left, a2, 0, ac_ip_a_left_ncol) * term_41
      out41c = out41c + Cpref * get2(aca_a_right, a0, a1, aca_a_right_ncol) * get2(ac_ip_a_left, a2, 0, ac_ip_a_left_ncol) * term_41

      out14x = out14x + Xpref * get2(aca_a_left, a0, a1, aca_a_left_ncol) * get2(ac_ip_a_right, a2, 0, ac_ip_a_right_ncol) * term_14
      out14c = out14c + Cpref * get2(aca_a_left, a0, a1, aca_a_left_ncol) * get2(ac_ip_a_right, a2, 0, ac_ip_a_right_ncol) * term_14

      out41x_2s = out41x_2s + Xpref * get2(ac_ip_a_left, a0, 0, ac_ip_a_left_ncol) &
           & * get2(sc_a_ba, idxB, a1, sc_a_ba_ncol) * get2(cs_ea_b, 0, a2, cs_ea_b_ncol) * value
      out41c_2s = out41c_2s + Cpref * get2(ac_ip_a_left, a0, 0, ac_ip_a_left_ncol) &
           & * get2(sc_a_ba, idxB, a1, sc_a_ba_ncol) * get2(cs_ea_b, 0, a2, cs_ea_b_ncol) * value

      out23x_2s_ip = out23x_2s_ip + Xpref * get2(sc_ip_b, a0, 0, sc_ip_b_ncol) * get2(cs_b_ba, idxB, a1, cs_b_ba_ncol) &
           & * get2(ca_ea_a_right, 0, a2, ca_ea_a_right_ncol) * value
      out23c_2s_ip = out23c_2s_ip + Cpref * get2(sc_ip_b, a0, 0, sc_ip_b_ncol) * get2(cs_b_ba, idxB, a1, cs_b_ba_ncol) &
           & * get2(ca_ea_a_right, 0, a2, ca_ea_a_right_ncol) * value

      out13x_2s_ip = out13x_2s_ip + Xpref * get2(sc_ip_b, a0, 0, sc_ip_b_ncol) * get2(cs_a_ab, a1, idxB, cs_a_ab_ncol) &
           & * get2(ca_ea_a_right, 0, a2, ca_ea_a_right_ncol) * value
      out13c_2s_ip = out13c_2s_ip + Cpref * get2(sc_ip_b, a0, 0, sc_ip_b_ncol) * get2(cs_a_ab, a1, idxB, cs_a_ab_ncol) &
           & * get2(ca_ea_a_right, 0, a2, ca_ea_a_right_ncol) * value

      out31x = out31x + Xpref * get2(aca_a_right, a0, a1, aca_a_right_ncol) * get2(ca_ea_a_left, 0, a2, ca_ea_a_left_ncol) * term_31
      out31c = out31c + Cpref * get2(aca_a_right, a0, a1, aca_a_right_ncol) * get2(ca_ea_a_left, 0, a2, ca_ea_a_left_ncol) * term_31

      out13x = out13x + Xpref * get2(aca_a_left, a0, a1, aca_a_left_ncol) * get2(ca_ea_a_right, 0, a2, ca_ea_a_right_ncol) * term_13
      out13c = out13c + Cpref * get2(aca_a_left, a0, a1, aca_a_left_ncol) * get2(ca_ea_a_right, 0, a2, ca_ea_a_right_ncol) * term_13

      out43x_scip = out43x_scip + Xpref * get2(ca_ea_a_right, 0, a0, ca_ea_a_right_ncol) &
           & * get2(sc_ip_b, a1, 0, sc_ip_b_ncol) * get2(ac_ip_a_left, a2, 0, ac_ip_a_left_ncol) * term_43_ea
      out43c_scip = out43c_scip + Cpref * get2(ca_ea_a_right, 0, a0, ca_ea_a_right_ncol) &
           & * get2(sc_ip_b, a1, 0, sc_ip_b_ncol) * get2(ac_ip_a_left, a2, 0, ac_ip_a_left_ncol) * term_43_ea

      out43c_csea = out43c_csea + Cpref * get2(ca_ea_a_right, 0, a0, ca_ea_a_right_ncol) &
           & * get2(cs_ea_b, 0, a1, cs_ea_b_ncol) * get2(ac_ip_a_left, a2, 0, ac_ip_a_left_ncol) * term_43_ip
      out43x_csea = out43x_csea + Xpref * get2(ca_ea_a_right, 0, a0, ca_ea_a_right_ncol) &
           & * get2(cs_ea_b, 0, a1, cs_ea_b_ncol) * get2(ac_ip_a_left, a2, 0, ac_ip_a_left_ncol) * term_43_ip

      call accum_new_u2(idxA, idxB, value)
    end subroutine accum_u2

    subroutine accum_u3(idxA, idxB, value, is_t1)
      integer, intent(in) :: idxA(3), idxB
      double precision, intent(in) :: value
      logical, intent(in) :: is_t1

      integer :: a, b0, c0, tA, tB
      integer :: p(3,3)  ! perms index, 1..3,1..3
      integer :: a0,a1,a2
      double precision :: term_31, term_41, term_13, term_14, term_43_ea, term_43_ip

      ! Build perms as in Python calc_term_BAAA_3
      a  = idxA(1)
      b0 = idxA(2)
      c0 = idxA(3)

      ! default perms:
      ! p2 = (a,b,c)  => stored as row3
      ! p0 = (b,c,a)  => row1
      ! p1 = (b,a,c)  => row2
      if (a == c0) then
        tA = a
        tB = b0
        a  = tB
        b0 = tA
        ! now:
        ! p0 = (a,b,c)  where (a,b,c) is (idxA2, idxA1, idxA3)
        p(1,1)=a;  p(1,2)=b0; p(1,3)=c0
        p(3,1)=b0; p(3,2)=c0; p(3,3)=a
        p(2,1)=b0; p(2,2)=a;  p(2,3)=c0
      else
        p(1,1)=b0; p(1,2)=c0; p(1,3)=a
        p(2,1)=b0; p(2,2)=a;  p(2,3)=c0
        p(3,1)=a;  p(3,2)=b0; p(3,3)=c0
      end if

      term_31    = value * get2(ac_ip_b_left,  idxB, 0, ac_ip_b_left_ncol)
      term_41    = value * get2(ca_ea_b_left,  0, idxB, ca_ea_b_left_ncol)
      term_13    = value * get2(ac_ip_b_right, idxB, 0, ac_ip_b_right_ncol)
      term_14    = value * get2(ca_ea_b_right, 0, idxB, ca_ea_b_right_ncol)
      term_43_ea = value * get2(ca_ea_b_left,  0, idxB, ca_ea_b_left_ncol)
      term_43_ip = value * get2(ac_ip_b_right, idxB, 0, ac_ip_b_right_ncol)

      if (is_t1) then
        ! Direct transcription of the Python is_type_1 branch.
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out12x = out12x + Xpref * get2(aca_a_left, a0,a1,aca_a_left_ncol) * get2(sc_b_ab,a2,idxB,sc_b_ab_ncol) * value
        out12c = out12c + Cpref * get2(aca_a_left, a0,a1,aca_a_left_ncol) * get2(sc_b_ab,a2,idxB,sc_b_ab_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out12x = out12x + Xpref * get2(aca_a_left, a0,a1,aca_a_left_ncol) * get2(sc_b_ab,a2,idxB,sc_b_ab_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out12c = out12c + Cpref * get2(aca_a_left, a0,a1,aca_a_left_ncol) * get2(sc_b_ab,a2,idxB,sc_b_ab_ncol) * value

        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out41x = out41x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_41
        out41c = out41c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_41

        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out41x = out41x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_41

        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out41c = out41c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_41

        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out14x = out14x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ac_ip_a_right,a2,0,ac_ip_a_right_ncol) * term_14
        out14c = out14c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ac_ip_a_right,a2,0,ac_ip_a_right_ncol) * term_14
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out14x = out14x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ac_ip_a_right,a2,0,ac_ip_a_right_ncol) * term_14
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out14c = out14c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ac_ip_a_right,a2,0,ac_ip_a_right_ncol) * term_14

        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out13x_2s_ip = out13x_2s_ip + Xpref * get2(sc_ip_b,a2,0,sc_ip_b_ncol) * get2(cs_a_ab,a0,idxB,cs_a_ab_ncol) &
             & * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out13x_2s_ip = out13x_2s_ip + Xpref * get2(sc_ip_b,a2,0,sc_ip_b_ncol) * get2(cs_a_ab,a0,idxB,cs_a_ab_ncol) &
             & * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) * value
        out13c_2s_ip = out13c_2s_ip + Cpref * get2(sc_ip_b,a2,0,sc_ip_b_ncol) * get2(cs_a_ab,a0,idxB,cs_a_ab_ncol) &
             & * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) * value             
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out13c_2s_ip = out13c_2s_ip + Cpref * get2(sc_ip_b,a2,0,sc_ip_b_ncol) * get2(cs_a_ab,a0,idxB,cs_a_ab_ncol) &
             & * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) * value

        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out23x_2s_ip = out23x_2s_ip + Xpref * get2(sc_ip_b,a1,0,sc_ip_b_ncol) * get2(cs_b_ba,idxB,a0,cs_b_ba_ncol) &
             & * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * value
        out23c_2s_ip = out23c_2s_ip + Cpref * get2(sc_ip_b,a1,0,sc_ip_b_ncol) * get2(cs_b_ba,idxB,a0,cs_b_ba_ncol) &
             & * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * value
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out23x_2s_ip = out23x_2s_ip + Xpref * get2(sc_ip_b,a1,0,sc_ip_b_ncol) * get2(cs_b_ba,idxB,a0,cs_b_ba_ncol) &
             & * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out23c_2s_ip = out23c_2s_ip + Cpref * get2(sc_ip_b,a1,0,sc_ip_b_ncol) * get2(cs_b_ba,idxB,a0,cs_b_ba_ncol) &
             & * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * value

        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out42x = out42x + Xpref * get2(sc_b_ab,a0,idxB,sc_b_ab_ncol) * get2(cs_ea_b,0,a1,cs_ea_b_ncol) &
             & * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out42c = out42c + Cpref * get2(sc_b_ab,a0,idxB,sc_b_ab_ncol) * get2(cs_ea_b,0,a1,cs_ea_b_ncol) &
             & * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * value
        out42x = out42x + Xpref * get2(sc_b_ab,a0,idxB,sc_b_ab_ncol) * get2(cs_ea_b,0,a1,cs_ea_b_ncol) &
             & * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out42c = out42c + Cpref * get2(sc_b_ab,a0,idxB,sc_b_ab_ncol) * get2(cs_ea_b,0,a1,cs_ea_b_ncol) &
             & * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * value

        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out21x = out21x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(cs_b_ba,idxB,a2,cs_b_ba_ncol) * value
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out21c = out21c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(cs_b_ba,idxB,a2,cs_b_ba_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out21x = out21x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(cs_b_ba,idxB,a2,cs_b_ba_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out21c = out21c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(cs_b_ba,idxB,a2,cs_b_ba_ncol) * value

        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out31x = out31x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ca_ea_a_left,0,a2,ca_ea_a_left_ncol) * term_31
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out31c = out31c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ca_ea_a_left,0,a2,ca_ea_a_left_ncol) * term_31
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out31x = out31x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ca_ea_a_left,0,a2,ca_ea_a_left_ncol) * term_31
        out31c = out31c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ca_ea_a_left,0,a2,ca_ea_a_left_ncol) * term_31

        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out13x = out13x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * term_13
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out13c = out13c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * term_13
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out13x = out13x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * term_13
        out13c = out13c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * term_13

        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out41x_2s = out41x_2s + Xpref * get2(ac_ip_a_left,a1,0,ac_ip_a_left_ncol) * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) &
             & * get2(cs_ea_b,0,a0,cs_ea_b_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out41c_2s = out41c_2s + Cpref * get2(ac_ip_a_left,a1,0,ac_ip_a_left_ncol) * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) &
             & * get2(cs_ea_b,0,a2,cs_ea_b_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out41x_2s = out41x_2s + Xpref * get2(ac_ip_a_left,a0,0,ac_ip_a_left_ncol) * get2(sc_a_ba,idxB,a1,sc_a_ba_ncol) &
             & * get2(cs_ea_b,0,a2,cs_ea_b_ncol) * value
        out41c_2s = out41c_2s + Cpref * get2(ac_ip_a_left,a1,0,ac_ip_a_left_ncol) * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) &
             & * get2(cs_ea_b,0,a2,cs_ea_b_ncol) * value

        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out43x_scip = out43x_scip + Xpref * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) &
             & * get2(sc_ip_b,a0,0,sc_ip_b_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_43_ea
        out43c_scip = out43c_scip + Cpref * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) &
             & * get2(sc_ip_b,a0,0,sc_ip_b_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_43_ea
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out43x_scip = out43x_scip + Xpref * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) &
             & * get2(sc_ip_b,a0,0,sc_ip_b_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_43_ea
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out43c_scip = out43c_scip + Cpref * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) &
             & * get2(sc_ip_b,a0,0,sc_ip_b_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_43_ea

        ! perms[1] contributes to BOTH C and X
        out43c_csea = out43c_csea + Cpref * get2(ca_ea_a_right, 0, p(2,1), ca_ea_a_right_ncol) &
             & * get2(cs_ea_b, 0, p(2,2), cs_ea_b_ncol) &
             & * get2(ac_ip_a_left, p(2,3), 0, ac_ip_a_left_ncol) * term_43_ip
        out43x_csea = out43x_csea + Xpref * get2(ca_ea_a_right, 0, p(2,1), ca_ea_a_right_ncol) &
             & * get2(cs_ea_b, 0, p(2,2), cs_ea_b_ncol) &
             & * get2(ac_ip_a_left, p(2,3), 0, ac_ip_a_left_ncol) * term_43_ip
        ! extra C term from perms[0]
        out43c_csea = out43c_csea + Cpref * get2(ca_ea_a_right, 0, p(1,1), ca_ea_a_right_ncol) &
             & * get2(cs_ea_b, 0, p(1,2), cs_ea_b_ncol) &
             & * get2(ac_ip_a_left, p(1,3), 0, ac_ip_a_left_ncol) * term_43_ip
        ! extra X term from perms[2]
        out43x_csea = out43x_csea + Xpref * get2(ca_ea_a_right, 0, p(3,1), ca_ea_a_right_ncol) &
             & * get2(cs_ea_b, 0, p(3,2), cs_ea_b_ncol) &
             & * get2(ac_ip_a_left, p(3,3), 0, ac_ip_a_left_ncol) * term_43_ip
      else
        ! Direct transcription of the Python else branch.
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out21x = out21x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(cs_b_ba,idxB,a2,cs_b_ba_ncol) * value
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out21c = out21c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(cs_b_ba,idxB,a2,cs_b_ba_ncol) * value

        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out12x = out12x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(sc_b_ab,a2,idxB,sc_b_ab_ncol) * value
        a0=p(3,2); a1=p(3,3); a2=p(3,1)
        out12c = out12c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(sc_b_ab,a2,idxB,sc_b_ab_ncol) * value

        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out31c = out31c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ca_ea_a_left,0,a2,ca_ea_a_left_ncol) * term_31
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out31x = out31x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ca_ea_a_left,0,a2,ca_ea_a_left_ncol) * term_31

        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out13c = out13c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * term_13
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out13x = out13x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * term_13

        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out41c_2s = out41c_2s + Cpref * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) &
             & * get2(cs_ea_b,0,a1,cs_ea_b_ncol) * value
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out41x_2s = out41x_2s + Xpref * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) &
             & * get2(cs_ea_b,0,a1,cs_ea_b_ncol) * value

        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out23c_2s_ip = out23c_2s_ip + Cpref * get2(sc_ip_b,a1,0,sc_ip_b_ncol) * get2(cs_b_ba,idxB,a0,cs_b_ba_ncol) &
             & * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out23x_2s_ip = out23x_2s_ip + Xpref * get2(sc_ip_b,a1,0,sc_ip_b_ncol) * get2(cs_b_ba,idxB,a0,cs_b_ba_ncol) &
             & * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * value

        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out41x = out41x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_41
        a0=p(3,2); a1=p(3,3); a2=p(3,1)
        out41c = out41c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_41

        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out14x = out14x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ac_ip_a_right,a2,0,ac_ip_a_right_ncol) * term_14
        a0=p(3,2); a1=p(3,3); a2=p(3,1)
        out14c = out14c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ac_ip_a_right,a2,0,ac_ip_a_right_ncol) * term_14

        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out13x_2s_ip = out13x_2s_ip + Xpref * get2(sc_ip_b,a2,0,sc_ip_b_ncol) * get2(cs_a_ab,a0,idxB,cs_a_ab_ncol) &
             & * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out13c_2s_ip = out13c_2s_ip + Cpref * get2(sc_ip_b,a2,0,sc_ip_b_ncol) * get2(cs_a_ab,a0,idxB,cs_a_ab_ncol) &
             & * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) * value

        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out42x = out42x + Xpref * get2(sc_b_ab,a0,idxB,sc_b_ab_ncol) * get2(cs_ea_b,0,a1,cs_ea_b_ncol) &
             & * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out42c = out42c + Cpref * get2(sc_b_ab,a0,idxB,sc_b_ab_ncol) * get2(cs_ea_b,0,a1,cs_ea_b_ncol) &
             & * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * value

        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out43x_scip = out43x_scip + Xpref * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) &
             & * get2(sc_ip_b,a0,0,sc_ip_b_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_43_ea
        a0=p(3,3); a1=p(3,2); a2=p(3,1)
        out43c_scip = out43c_scip + Cpref * get2(ca_ea_a_right,0,a0,ca_ea_a_right_ncol) &
             & * get2(sc_ip_b,a1,0,sc_ip_b_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_43_ea

        ! X: perms[2]
        out43x_csea = out43x_csea + Xpref * get2(ca_ea_a_right, 0, p(3,1), ca_ea_a_right_ncol) &
             & * get2(cs_ea_b, 0, p(3,2), cs_ea_b_ncol) &
             & * get2(ac_ip_a_left, p(3,3), 0, ac_ip_a_left_ncol) * term_43_ip
        
        ! C: perms[0]
        out43c_csea = out43c_csea + Cpref * get2(ca_ea_a_right, 0, p(1,1), ca_ea_a_right_ncol) &
             & * get2(cs_ea_b, 0, p(1,2), cs_ea_b_ncol) &
             & * get2(ac_ip_a_left, p(1,3), 0, ac_ip_a_left_ncol) * term_43_ip
      end if

      call accum_new_u3(p, idxB, value, is_t1)
    end subroutine accum_u3

    subroutine accum_u4(idxA, idxB, value)
      integer, intent(in) :: idxA(3), idxB
      double precision, intent(in) :: value
      integer :: i0,i1,i2
      integer :: p(6,3)
      integer :: ii, a0,a1,a2
      double precision :: term_31, term_41, term_13, term_14, term_43_ea, term_43_ip

      i0 = idxA(1); i1 = idxA(2); i2 = idxA(3)

      ! perms = [p0, p1, p2, p3, p4, p5]
      ! p5=(i0,i1,i2); p3=(i0,i2,i1); p4=(i1,i0,i2); p1=(i1,i2,i0); p2=(i2,i0,i1); p0=(i2,i1,i0)
      p(6,1)=i0; p(6,2)=i1; p(6,3)=i2
      p(4,1)=i0; p(4,2)=i2; p(4,3)=i1
      p(5,1)=i1; p(5,2)=i0; p(5,3)=i2
      p(2,1)=i1; p(2,2)=i2; p(2,3)=i0
      p(3,1)=i2; p(3,2)=i0; p(3,3)=i1
      p(1,1)=i2; p(1,2)=i1; p(1,3)=i0

      term_31    = value * get2(ac_ip_b_left,  idxB, 0, ac_ip_b_left_ncol)
      term_41    = value * get2(ca_ea_b_left,  0, idxB, ca_ea_b_left_ncol)
      term_13    = value * get2(ac_ip_b_right, idxB, 0, ac_ip_b_right_ncol)
      term_14    = value * get2(ca_ea_b_right, 0, idxB, ca_ea_b_right_ncol)
      term_43_ea = value * get2(ca_ea_b_left,  0, idxB, ca_ea_b_left_ncol)
      term_43_ip = value * get2(ac_ip_b_right, idxB, 0, ac_ip_b_right_ncol)

      ! idx in [2,4] => perms[2], perms[4] (1-based: 3,5)
      do ii = 3, 5, 2
        a0=p(ii,1); a1=p(ii,2); a2=p(ii,3)
        out41x = out41x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_41
        out14x = out14x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ac_ip_a_right,a2,0,ac_ip_a_right_ncol) * term_14
        out43x_scip = out43x_scip + Xpref * get2(sc_ip_b,a0,0,sc_ip_b_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) &
             & * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) * term_43_ea
        out13x_2s_ip = out13x_2s_ip + Xpref * get2(cs_a_ab,a0,idxB,cs_a_ab_ncol) &
             & * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) * get2(sc_ip_b,a2,0,sc_ip_b_ncol) * value
        out42x = out42x + Xpref * get2(sc_b_ab,a0,idxB,sc_b_ab_ncol) * get2(cs_ea_b,0,a1,cs_ea_b_ncol) &
             & * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * value
        out12x = out12x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(sc_b_ab,a2,idxB,sc_b_ab_ncol) * value
      end do

      ! idx in [0,1] => perms[0], perms[1] (1-based: 1,2)
      do ii = 1, 2
        a0=p(ii,1); a1=p(ii,2); a2=p(ii,3)
        out41c = out41c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_41
        out14c = out14c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ac_ip_a_right,a2,0,ac_ip_a_right_ncol) * term_14
        out12c = out12c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(sc_b_ab,a2,idxB,sc_b_ab_ncol) * value
        out31x = out31x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ca_ea_a_left,0,a2,ca_ea_a_left_ncol) * term_31
        out13x = out13x + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * term_13
        out41x_2s = out41x_2s + Xpref * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) &
             & * get2(cs_ea_b,0,a1,cs_ea_b_ncol) * value
        out23x_2s_ip = out23x_2s_ip + Xpref * get2(sc_ip_b,a2,0,sc_ip_b_ncol) * get2(cs_b_ba,idxB,a1,cs_b_ba_ncol) &
             & * get2(ca_ea_a_right,0,a0,ca_ea_a_right_ncol) * value
        out43c_scip = out43c_scip + Cpref * get2(sc_ip_b,a0,0,sc_ip_b_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) &
             & * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) * term_43_ea
        out43c_csea = out43c_csea + Cpref * get2(ca_ea_a_right,0,a0,ca_ea_a_right_ncol) * get2(cs_ea_b,0,a1,cs_ea_b_ncol) &
             & * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_43_ip
        out21c = out21c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(cs_b_ba,idxB,a2,cs_b_ba_ncol) * value
      end do

      ! idx in [3,5] => perms[3], perms[5] (1-based: 4,6)
      do ii = 4, 6, 2
        a0=p(ii,1); a1=p(ii,2); a2=p(ii,3)
        out31c = out31c + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(ca_ea_a_left,0,a2,ca_ea_a_left_ncol) * term_31
        out13c = out13c + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * term_13
        out41c_2s = out41c_2s + Cpref * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) &
             & * get2(cs_ea_b,0,a1,cs_ea_b_ncol) * value
        out13c_2s_ip = out13c_2s_ip + Cpref * get2(cs_a_ab,a0,idxB,cs_a_ab_ncol) * get2(ca_ea_a_right,0,a1,ca_ea_a_right_ncol) &
             & * get2(sc_ip_b,a2,0,sc_ip_b_ncol) * value
        out42c = out42c + Cpref * get2(sc_b_ab,a0,idxB,sc_b_ab_ncol) * get2(cs_ea_b,0,a1,cs_ea_b_ncol) &
             & * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * value
        out23c_2s_ip = out23c_2s_ip + Cpref * get2(cs_b_ba,idxB,a0,cs_b_ba_ncol) * get2(sc_ip_b,a1,0,sc_ip_b_ncol) &
             & * get2(ca_ea_a_right,0,a2,ca_ea_a_right_ncol) * value
        out43x_csea = out43x_csea + Xpref * get2(ca_ea_a_right,0,a0,ca_ea_a_right_ncol) &
             & * get2(cs_ea_b,0,a1,cs_ea_b_ncol) * get2(ac_ip_a_left,a2,0,ac_ip_a_left_ncol) * term_43_ip
        out21x = out21x + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) * get2(cs_b_ba,idxB,a2,cs_b_ba_ncol) * value
      end do

      call accum_new_u4(p, idxB, value)
    end subroutine accum_u4

    subroutine accum_new_u2(idxA, idxB, value)
      integer, intent(in) :: idxA(3), idxB
      double precision, intent(in) :: value
      integer :: a0, a1, a2

      a0 = idxA(1); a1 = idxA(2); a2 = idxA(3)

      out14c_2s_virt_ea_arb = out14c_2s_virt_ea_arb + Cpref * dot_ipr_eabq(a0,a1) &
           & * get2(cs_a_ab,a2,idxB,cs_a_ab_ncol) * value
      out14x_2s_virt_ea_arb = out14x_2s_virt_ea_arb + Xpref * dot_ipr_eabq(a0,a1) &
           & * get2(cs_a_ab,a2,idxB,cs_a_ab_ncol) * value

      out21c_3s_b_virt_b_occ_arb_a_virt_arb = out21c_3s_b_virt_b_occ_arb_a_virt_arb + Cpref &
           & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,a2,idxB,cs_a_ab_iq_ncol) * value
      out21x_3s_b_virt_b_occ_arb_a_virt_arb = out21x_3s_b_virt_b_occ_arb_a_virt_arb + Xpref &
           & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,a2,idxB,cs_a_ab_iq_ncol) * value

      out21c_3s_a_occ_b_virt_b_occ_arb = out21c_3s_a_occ_b_virt_b_occ_arb + Cpref &
           & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value
      out21x_3s_a_occ_b_virt_b_occ_arb = out21x_3s_a_occ_b_virt_b_occ_arb + Xpref &
           & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value

      out14c_2s_ip_ea_arb = out14c_2s_ip_ea_arb + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) &
           & * dot_eabq_ipa(a2,idxB) * value
      out14x_2s_ip_ea_arb = out14x_2s_ip_ea_arb + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) &
           & * dot_eabq_ipa(a2,idxB) * value

      out14c_2s_occ_arb_ea_arb = out14c_2s_occ_arb_ea_arb + Cpref * dot_ipr_eabq(a0,a1) &
           & * get2(sc_a_ba_pa,idxB,a2,sc_a_ba_pa_ncol) * value
      out14x_2s_occ_arb_ea_arb = out14x_2s_occ_arb_ea_arb + Xpref * dot_ipr_eabq(a0,a1) &
           & * get2(sc_a_ba_pa,idxB,a2,sc_a_ba_pa_ncol) * value

      out31c_2s_occ_ip_arb = out31c_2s_occ_ip_arb + Cpref * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) &
           & * dot_eala_ipbp(a1,a2) * value
      out31x_2s_occ_ip_arb = out31x_2s_occ_ip_arb + Xpref * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) &
           & * dot_eala_ipbp(a1,a2) * value

      out31c_2s_ea_ip_arb = out31c_2s_ea_ip_arb + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
           & * dot_ipbp_eaa(a2,idxB) * value
      out31x_2s_ea_ip_arb = out31x_2s_ea_ip_arb + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
           & * dot_ipbp_eaa(a2,idxB) * value

      out21c_1s_occ_arb = out21c_1s_occ_arb + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
           & * get2(sc_b_ab_pa,a2,idxB,sc_b_ab_pa_ncol) * value
      out21x_1s_occ_arb = out21x_1s_occ_arb + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
           & * get2(sc_b_ab_pa,a2,idxB,sc_b_ab_pa_ncol) * value
    end subroutine accum_new_u2

    subroutine accum_new_u3(p, idxB, value, is_t1)
      integer, intent(in) :: p(3,3), idxB
      double precision, intent(in) :: value
      logical, intent(in) :: is_t1
      integer :: a0, a1, a2

      if (is_t1) then
        ! 14: BAAA 2S virt/EA(arb).  Python order: C p1, X p0, C p0, X p2.
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out14c_2s_virt_ea_arb = out14c_2s_virt_ea_arb + Cpref * dot_ipr_eabq(a0,a1) &
             & * get2(cs_a_ab,a2,idxB,cs_a_ab_ncol) * value
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out14x_2s_virt_ea_arb = out14x_2s_virt_ea_arb + Xpref * dot_ipr_eabq(a0,a1) &
             & * get2(cs_a_ab,a2,idxB,cs_a_ab_ncol) * value
        out14c_2s_virt_ea_arb = out14c_2s_virt_ea_arb + Cpref * dot_ipr_eabq(a0,a1) &
             & * get2(cs_a_ab,a2,idxB,cs_a_ab_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out14x_2s_virt_ea_arb = out14x_2s_virt_ea_arb + Xpref * dot_ipr_eabq(a0,a1) &
             & * get2(cs_a_ab,a2,idxB,cs_a_ab_ncol) * value

        ! 21: AAAB 1S occ(arb).  Same permutation pattern as the existing 12 term.
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out21x_1s_occ_arb = out21x_1s_occ_arb + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * get2(sc_b_ab_pa,a2,idxB,sc_b_ab_pa_ncol) * value
        out21c_1s_occ_arb = out21c_1s_occ_arb + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * get2(sc_b_ab_pa,a2,idxB,sc_b_ab_pa_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out21x_1s_occ_arb = out21x_1s_occ_arb + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * get2(sc_b_ab_pa,a2,idxB,sc_b_ab_pa_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out21c_1s_occ_arb = out21c_1s_occ_arb + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * get2(sc_b_ab_pa,a2,idxB,sc_b_ab_pa_ncol) * value

        ! 31: BAAA 2S EA/IP(arb).
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out31x_2s_ea_ip_arb = out31x_2s_ea_ip_arb + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * dot_ipbp_eaa(a2,idxB) * value
        out31c_2s_ea_ip_arb = out31c_2s_ea_ip_arb + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * dot_ipbp_eaa(a2,idxB) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out31x_2s_ea_ip_arb = out31x_2s_ea_ip_arb + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * dot_ipbp_eaa(a2,idxB) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out31c_2s_ea_ip_arb = out31c_2s_ea_ip_arb + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * dot_ipbp_eaa(a2,idxB) * value

        ! 21: ABAA 3S B-virt/B-occ(arb)/A-virt(arb).
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out21x_3s_b_virt_b_occ_arb_a_virt_arb = out21x_3s_b_virt_b_occ_arb_a_virt_arb + Xpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,a2,idxB,cs_a_ab_iq_ncol) * value
        out21c_3s_b_virt_b_occ_arb_a_virt_arb = out21c_3s_b_virt_b_occ_arb_a_virt_arb + Cpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,a2,idxB,cs_a_ab_iq_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out21x_3s_b_virt_b_occ_arb_a_virt_arb = out21x_3s_b_virt_b_occ_arb_a_virt_arb + Xpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,a2,idxB,cs_a_ab_iq_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out21c_3s_b_virt_b_occ_arb_a_virt_arb = out21c_3s_b_virt_b_occ_arb_a_virt_arb + Cpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,a2,idxB,cs_a_ab_iq_ncol) * value

        ! 14: AABA 2S occ(arb)/EA(arb).
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out14x_2s_occ_arb_ea_arb = out14x_2s_occ_arb_ea_arb + Xpref * dot_ipr_eabq(a0,a1) &
             & * get2(sc_a_ba_pa,idxB,a2,sc_a_ba_pa_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out14c_2s_occ_arb_ea_arb = out14c_2s_occ_arb_ea_arb + Cpref * dot_ipr_eabq(a0,a1) &
             & * get2(sc_a_ba_pa,idxB,a2,sc_a_ba_pa_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out14x_2s_occ_arb_ea_arb = out14x_2s_occ_arb_ea_arb + Xpref * dot_ipr_eabq(a0,a1) &
             & * get2(sc_a_ba_pa,idxB,a2,sc_a_ba_pa_ncol) * value
        out14c_2s_occ_arb_ea_arb = out14c_2s_occ_arb_ea_arb + Cpref * dot_ipr_eabq(a0,a1) &
             & * get2(sc_a_ba_pa,idxB,a2,sc_a_ba_pa_ncol) * value

        ! 21: ABAA 3S A-occ/B-virt/B-occ(arb).
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out21x_3s_a_occ_b_virt_b_occ_arb = out21x_3s_a_occ_b_virt_b_occ_arb + Xpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out21x_3s_a_occ_b_virt_b_occ_arb = out21x_3s_a_occ_b_virt_b_occ_arb + Xpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out21c_3s_a_occ_b_virt_b_occ_arb = out21c_3s_a_occ_b_virt_b_occ_arb + Cpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out21c_3s_a_occ_b_virt_b_occ_arb = out21c_3s_a_occ_b_virt_b_occ_arb + Cpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value

        ! 14: ABAA 2S IP/EA(arb).
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out14x_2s_ip_ea_arb = out14x_2s_ip_ea_arb + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) &
             & * dot_eabq_ipa(a2,idxB) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out14c_2s_ip_ea_arb = out14c_2s_ip_ea_arb + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) &
             & * dot_eabq_ipa(a2,idxB) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out14x_2s_ip_ea_arb = out14x_2s_ip_ea_arb + Xpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) &
             & * dot_eabq_ipa(a2,idxB) * value
        out14c_2s_ip_ea_arb = out14c_2s_ip_ea_arb + Cpref * get2(aca_a_left,a0,a1,aca_a_left_ncol) &
             & * dot_eabq_ipa(a2,idxB) * value

        ! 31: ABAA 2S occ/IP(arb); IP(arb) is on p(...,1) in this branch.
        a0=p(1,1); a1=p(1,2); a2=p(1,3)
        out31x_2s_occ_ip_arb = out31x_2s_occ_ip_arb + Xpref * dot_eala_ipbp(a1,a0) &
             & * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value
        a0=p(3,1); a1=p(3,2); a2=p(3,3)
        out31c_2s_occ_ip_arb = out31c_2s_occ_ip_arb + Cpref * dot_eala_ipbp(a1,a0) &
             & * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value
        a0=p(2,1); a1=p(2,2); a2=p(2,3)
        out31x_2s_occ_ip_arb = out31x_2s_occ_ip_arb + Xpref * dot_eala_ipbp(a1,a0) &
             & * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value
        out31c_2s_occ_ip_arb = out31c_2s_occ_ip_arb + Cpref * dot_eala_ipbp(a1,a0) &
             & * get2(sc_a_ba,idxB,a2,sc_a_ba_ncol) * value
      else
        ! Direct transcription of the Python unique-count-3 non-type-1 branch.
        out14c_2s_virt_ea_arb = out14c_2s_virt_ea_arb + Cpref * dot_ipr_eabq(p(3,2),p(3,3)) &
             & * get2(cs_a_ab,p(3,1),idxB,cs_a_ab_ncol) * value
        out14x_2s_virt_ea_arb = out14x_2s_virt_ea_arb + Xpref * dot_ipr_eabq(p(2,1),p(2,2)) &
             & * get2(cs_a_ab,p(2,3),idxB,cs_a_ab_ncol) * value

        out21x_1s_occ_arb = out21x_1s_occ_arb + Xpref * get2(aca_a_right,p(2,1),p(2,2),aca_a_right_ncol) &
             & * get2(sc_b_ab_pa,p(2,3),idxB,sc_b_ab_pa_ncol) * value
        out21c_1s_occ_arb = out21c_1s_occ_arb + Cpref * get2(aca_a_right,p(3,2),p(3,3),aca_a_right_ncol) &
             & * get2(sc_b_ab_pa,p(3,1),idxB,sc_b_ab_pa_ncol) * value

        out14c_2s_occ_arb_ea_arb = out14c_2s_occ_arb_ea_arb + Cpref * dot_ipr_eabq(p(3,1),p(3,2)) &
             & * get2(sc_a_ba_pa,idxB,p(3,3),sc_a_ba_pa_ncol) * value
        out14x_2s_occ_arb_ea_arb = out14x_2s_occ_arb_ea_arb + Xpref * dot_ipr_eabq(p(1,1),p(1,2)) &
             & * get2(sc_a_ba_pa,idxB,p(1,3),sc_a_ba_pa_ncol) * value

        out21c_3s_a_occ_b_virt_b_occ_arb = out21c_3s_a_occ_b_virt_b_occ_arb + Cpref &
             & * get2(scs_b_aa_pa,p(3,1),p(3,2),scs_b_aa_pa_ncol) * get2(sc_a_ba,idxB,p(3,3),sc_a_ba_ncol) * value
        out21x_3s_a_occ_b_virt_b_occ_arb = out21x_3s_a_occ_b_virt_b_occ_arb + Xpref &
             & * get2(scs_b_aa_pa,p(1,1),p(1,2),scs_b_aa_pa_ncol) * get2(sc_a_ba,idxB,p(1,3),sc_a_ba_ncol) * value

        out14c_2s_ip_ea_arb = out14c_2s_ip_ea_arb + Cpref * get2(aca_a_left,p(3,1),p(3,2),aca_a_left_ncol) &
             & * dot_eabq_ipa(p(3,3),idxB) * value
        out14x_2s_ip_ea_arb = out14x_2s_ip_ea_arb + Xpref * get2(aca_a_left,p(1,1),p(1,2),aca_a_left_ncol) &
             & * dot_eabq_ipa(p(1,3),idxB) * value

        out31c_2s_occ_ip_arb = out31c_2s_occ_ip_arb + Cpref * dot_eala_ipbp(p(3,2),p(3,1)) &
             & * get2(sc_a_ba,idxB,p(3,3),sc_a_ba_ncol) * value
        out31x_2s_occ_ip_arb = out31x_2s_occ_ip_arb + Xpref * dot_eala_ipbp(p(1,2),p(1,1)) &
             & * get2(sc_a_ba,idxB,p(1,3),sc_a_ba_ncol) * value

        out31x_2s_ea_ip_arb = out31x_2s_ea_ip_arb + Xpref * get2(aca_a_right,p(2,1),p(2,2),aca_a_right_ncol) &
             & * dot_ipbp_eaa(p(2,3),idxB) * value
        out31c_2s_ea_ip_arb = out31c_2s_ea_ip_arb + Cpref * get2(aca_a_right,p(3,2),p(3,3),aca_a_right_ncol) &
             & * dot_ipbp_eaa(p(3,1),idxB) * value

        out21c_3s_b_virt_b_occ_arb_a_virt_arb = out21c_3s_b_virt_b_occ_arb_a_virt_arb + Cpref &
             & * get2(scs_b_aa_pa,p(3,2),p(3,3),scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,p(3,1),idxB,cs_a_ab_iq_ncol) * value
        out21x_3s_b_virt_b_occ_arb_a_virt_arb = out21x_3s_b_virt_b_occ_arb_a_virt_arb + Xpref &
             & * get2(scs_b_aa_pa,p(2,1),p(2,2),scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,p(2,3),idxB,cs_a_ab_iq_ncol) * value
      end if
    end subroutine accum_new_u3

    subroutine accum_new_u4(p, idxB, value)
      integer, intent(in) :: p(6,3), idxB
      double precision, intent(in) :: value
      integer :: ii, a0, a1, a2

      ! Python idx in [2,4]: exchange-only group for these four terms.
      do ii = 3, 5, 2
        a0=p(ii,1); a1=p(ii,2); a2=p(ii,3)
        out14x_2s_virt_ea_arb = out14x_2s_virt_ea_arb + Xpref * dot_ipr_eabq(a0,a1) &
             & * get2(cs_a_ab,a2,idxB,cs_a_ab_ncol) * value
        out31x_2s_ea_ip_arb = out31x_2s_ea_ip_arb + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * dot_ipbp_eaa(a2,idxB) * value
        out21x_3s_b_virt_b_occ_arb_a_virt_arb = out21x_3s_b_virt_b_occ_arb_a_virt_arb + Xpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,a2,idxB,cs_a_ab_iq_ncol) * value
        out21x_1s_occ_arb = out21x_1s_occ_arb + Xpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * get2(sc_b_ab_pa,a2,idxB,sc_b_ab_pa_ncol) * value
      end do

      ! Python idx in [0,1]: Coulomb group plus the exchange parts whose
      ! contraction pattern belongs to these permutations.
      do ii = 1, 2
        a0=p(ii,1); a1=p(ii,2); a2=p(ii,3)
        out14c_2s_virt_ea_arb = out14c_2s_virt_ea_arb + Cpref * dot_ipr_eabq(a0,a1) &
             & * get2(cs_a_ab,a2,idxB,cs_a_ab_ncol) * value
        out31c_2s_ea_ip_arb = out31c_2s_ea_ip_arb + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * dot_ipbp_eaa(a2,idxB) * value
        out21c_1s_occ_arb = out21c_1s_occ_arb + Cpref * get2(aca_a_right,a0,a1,aca_a_right_ncol) &
             & * get2(sc_b_ab_pa,a2,idxB,sc_b_ab_pa_ncol) * value
        out21c_3s_b_virt_b_occ_arb_a_virt_arb = out21c_3s_b_virt_b_occ_arb_a_virt_arb + Cpref &
             & * get2(scs_b_aa_pa,a0,a1,scs_b_aa_pa_ncol) * get2(cs_a_ab_iq,a2,idxB,cs_a_ab_iq_ncol) * value

        out14x_2s_occ_arb_ea_arb = out14x_2s_occ_arb_ea_arb + Xpref * dot_ipr_eabq(a0,a1) &
             & * get2(sc_a_ba_pa,idxB,a2,sc_a_ba_pa_ncol) * value
        out21c_3s_a_occ_b_virt_b_occ_arb = out21c_3s_a_occ_b_virt_b_occ_arb + Cpref &
             & * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) * get2(scs_b_aa_pa,a2,a1,scs_b_aa_pa_ncol) * value
        out14c_2s_ip_ea_arb = out14c_2s_ip_ea_arb + Cpref * get2(aca_a_left,a2,a1,aca_a_left_ncol) &
             & * dot_eabq_ipa(a0,idxB) * value
        out31c_2s_occ_ip_arb = out31c_2s_occ_ip_arb + Cpref * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) &
             & * dot_eala_ipbp(a1,a2) * value
      end do

      ! Python idx in [3,5].
      do ii = 4, 6, 2
        a0=p(ii,1); a1=p(ii,2); a2=p(ii,3)
        out14c_2s_occ_arb_ea_arb = out14c_2s_occ_arb_ea_arb + Cpref * dot_ipr_eabq(a0,a1) &
             & * get2(sc_a_ba_pa,idxB,a2,sc_a_ba_pa_ncol) * value
        out21x_3s_a_occ_b_virt_b_occ_arb = out21x_3s_a_occ_b_virt_b_occ_arb + Xpref &
             & * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) * get2(scs_b_aa_pa,a2,a1,scs_b_aa_pa_ncol) * value
        out14x_2s_ip_ea_arb = out14x_2s_ip_ea_arb + Xpref * get2(aca_a_left,a2,a1,aca_a_left_ncol) &
             & * dot_eabq_ipa(a0,idxB) * value
        out31x_2s_occ_ip_arb = out31x_2s_occ_ip_arb + Xpref * get2(sc_a_ba,idxB,a0,sc_a_ba_ncol) &
             & * dot_eala_ipbp(a1,a2) * value
      end do
    end subroutine accum_new_u4

  end subroutine baaa_accum_all

end module twoelint_baaa_mod
